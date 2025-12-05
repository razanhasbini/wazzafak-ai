package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"code.sajari.com/docconv"
	openai "github.com/sashabaranov/go-openai"
)

/* -------------------------------------------------------------------------- */
/*                    HTTP: /v1/upload-cv  (multipart POST)                   */
/* -------------------------------------------------------------------------- */
/*
  Multipart form fields:
    - user_id: number (MUST be an existing users.id)
    - file:    (pdf | docx | txt)

  Response:
    { "message": "...", "field": "<detected>", "analysis": "<feedback>" }

  Side effects:
    - Saves analysis to public.cv_analyses (analysis=text JSON, comments=field)
    - Also stores full CV text into ai.cv_documents and ai.cv_chunks (+ embeddings)
*/

func (a *App) handleUploadCV(w http.ResponseWriter, r *http.Request) {
	fmt.Println("📄 /v1/cv/upload hit")

	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(20 << 20); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	file, handler, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	userIDStr := strings.TrimSpace(r.FormValue("user_id"))
	if userIDStr == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}
	userID, err := strconv.Atoi(userIDStr)
	if err != nil || userID <= 0 {
		http.Error(w, "invalid user_id", http.StatusBadRequest)
		return
	}

	// Read file
	data, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "file read error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Extract text with docconv (requires pdftotext for PDFs)
	text, err := extractText(data, handler.Filename)
	if err != nil {
		http.Error(w, "text extraction error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	text = normalizeCVText(text)
	if len(strings.TrimSpace(text)) == 0 {
		http.Error(w, "no text extracted", http.StatusBadRequest)
		return
	}

	// Analyze with OpenAI
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	sysPrompt := `You are an AI career analyst. You will read a candidate's CV and identify:
1) "field": Software Engineering | Business | Mechanical Engineering | Graphic Design | Marketing | Other
2) "subfield": short (e.g., Frontend, Backend, Full Stack, Mobile, HR, Finance, Operations, Marketing, Automotive, Aerospace, Manufacturing, UI/UX, Branding, Illustration, Digital Marketing, Content Marketing, SEO)
3) "analysis": a short, personalized paragraph (2–4 sentences) focusing on strengths + gaps.

Return STRICT JSON:
{ "field": "...", "subfield": "...", "analysis": "..." }`

	resp, err := a.LLM.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.25,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sysPrompt},
			{Role: openai.ChatMessageRoleUser, Content: text},
		},
	})
	if err != nil {
		http.Error(w, "AI analysis failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	js := extractJSON(raw)

	var result struct {
		Field    string `json:"field"`
		Subfield string `json:"subfield"`
		Analysis string `json:"analysis"`
	}
	_ = json.Unmarshal([]byte(js), &result)
	if strings.TrimSpace(result.Field) == "" {
		result.Field = "Unknown"
	}
	// Persist analysis (keep schema unchanged)
	_, err = a.DB.Exec(`
    INSERT INTO cv_analyses (user_id, analysis, comments, created_at, updated_at)
    VALUES ($1, $2, $3, NOW(), NOW())`, userID, js, result.Field)
	if err != nil {
		http.Error(w, "DB insert error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Also store CV text + chunks + embeddings for retrieval during live interview
	if err := a.ingestCVForUser(r.Context(), int64(userID), text); err != nil {
		// Non-fatal: still return success for the upload/analyze
		fmt.Println("⚠️ ingestCVForUser:", err)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"message":  "CV uploaded and analyzed successfully",
		"field":    result.Field,
		"analysis": result.Analysis,
	})
}

/* -------------------------------------------------------------------------- */
/*                      Text Extraction (docconv + normalize)                 */
/* -------------------------------------------------------------------------- */

func extractText(data []byte, filename string) (string, error) {
	contentType := ""
	lfn := strings.ToLower(filename)
	switch {
	case strings.HasSuffix(lfn, ".pdf"):
		contentType = "application/pdf"
	case strings.HasSuffix(lfn, ".docx"):
		contentType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	case strings.HasSuffix(lfn, ".txt"):
		contentType = "text/plain"
	default:
		return "", fmt.Errorf("unsupported file type: %s", filename)
	}
	reader := bytes.NewReader(data)
	res, err := docconv.Convert(reader, contentType, false)
	if err != nil {
		return "", fmt.Errorf("docconv convert error: %w", err)
	}
	return strings.TrimSpace(res.Body), nil
}

func normalizeCVText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	sp := regexp.MustCompile(`[ \t]+`)
	s = sp.ReplaceAllString(s, " ")
	bl := regexp.MustCompile(`\n{2,}`)
	s = bl.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

/* -------------------------------------------------------------------------- */
/*                    Helper: ingest CV into ai.cv_* tables                   */
/* -------------------------------------------------------------------------- */

func (a *App) ingestCVForUser(ctx context.Context, userID int64, text string) error {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	tx, err := a.DB.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	var cvID int64
	if err := tx.QueryRow(`INSERT INTO ai.cv_documents(user_id,text) VALUES ($1,$2) RETURNING id`, userID, text).Scan(&cvID); err != nil {
		return err
	}

	chunks := chunkText(text, 4000, 400)
	embs, err := a.embed(ctx, chunks)
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO ai.cv_chunks(cv_id,chunk_text,embedding_json,ord) VALUES ($1,$2,$3::jsonb,$4)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for i := range chunks {
		if _, err := stmt.Exec(cvID, chunks[i], string(toJSON(embs[i])), i); err != nil {
			return err
		}
	}
	return tx.Commit()
}
