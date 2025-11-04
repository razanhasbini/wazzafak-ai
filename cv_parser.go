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
/*                          HTTP: /v1/upload-cv (POST)                        */
/* -------------------------------------------------------------------------- */
/*
  Multipart form expected:
    - user_id: number
    - file: file (pdf | docx | txt)

  Response:
    { "message": "...", "field": "<detected>", "analysis": "<feedback>" }
*/

func (a *App) handleUploadCV(w http.ResponseWriter, r *http.Request) {
	fmt.Println("📄 /v1/cv/upload hit")

	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(20 << 20)
	if err != nil {
		fmt.Println("❌ ParseMultipartForm:", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	file, handler, err := r.FormFile("file")
	if err != nil {
		fmt.Println("❌ FormFile error:", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	fmt.Println("📦 Uploaded file:", handler.Filename)

	userIDStr := strings.TrimSpace(r.FormValue("user_id"))
	fmt.Println("🧠 Received user_id (raw):", userIDStr)

	userID, err := strconv.Atoi(userIDStr)
	if err != nil {
		fmt.Println("❌ Invalid user_id (not an integer):", userIDStr)
		http.Error(w, "invalid user_id", http.StatusBadRequest)
		return
	}

	fmt.Printf("➡️ Parsed userID (int): %d\n", userID)

	// Read file into memory
	data, err := io.ReadAll(file)
	if err != nil {
		fmt.Println("❌ File read error:", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// 🧩 Extract text using docconv
	text, err := extractText(data, handler.Filename)
	if err != nil {
		fmt.Println("❌ Text extraction error:", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	text = normalizeCVText(text)
	if len(strings.TrimSpace(text)) == 0 {
		fmt.Println("❌ No text extracted")
		http.Error(w, "no text extracted", http.StatusBadRequest)
		return
	}

	// 🧠 Use OpenAI to classify
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	sysPrompt := `You are an AI career analyst. You will read a candidate's CV and identify:
1. The candidate's likely field (e.g., Software Engineering, Marketing, Business, Mechanical Engineering, Medicine, etc.).
2. A short analysis of their strengths and weaknesses.
Return JSON with keys "field" and "analysis".`

	resp, err := a.LLM.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.3,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sysPrompt},
			{Role: openai.ChatMessageRoleUser, Content: text},
		},
	})
	if err != nil {
		fmt.Println("❌ AI analysis failed:", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	var result struct {
		Field    string `json:"field"`
		Analysis string `json:"analysis"`
	}
	_ = json.Unmarshal([]byte(extractJSON(raw)), &result)

	if result.Field == "" {
		result.Field = "Unknown"
	}

	fmt.Printf("🧾 DEBUG — userID=%d | analysis='%s' | comments='%s'\n", userID, result.Analysis, result.Field)

	_, err = a.DB.Exec(`
    INSERT INTO cv_analyses (user_id, analysis, comments, created_at, updated_at)
    VALUES ($1, $2, $3, NOW(), NOW())
`, userID, result.Analysis, result.Field)

	if err != nil {
		fmt.Println("❌ DB insert error:", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	respData := map[string]string{
		"message":  "CV uploaded and analyzed successfully",
		"field":    result.Field,
		"analysis": result.Analysis,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(respData)

	fmt.Println("✅ CV processed successfully for user", userID)
}

/* -------------------------------------------------------------------------- */
/*                           Text Extraction (docconv)                        */
/* -------------------------------------------------------------------------- */

func extractText(data []byte, filename string) (string, error) {
	contentType := ""
	if strings.HasSuffix(strings.ToLower(filename), ".pdf") {
		contentType = "application/pdf"
	} else if strings.HasSuffix(strings.ToLower(filename), ".docx") {
		contentType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	} else if strings.HasSuffix(strings.ToLower(filename), ".txt") {
		contentType = "text/plain"
	} else {
		return "", fmt.Errorf("unsupported file type: %s", filename)
	}

	reader := bytes.NewReader(data)
	res, err := docconv.Convert(reader, contentType, true)
	if err != nil {
		return "", fmt.Errorf("docconv convert error: %w", err)
	}

	return strings.TrimSpace(res.Body), nil
}

/* -------------------------------------------------------------------------- */
/*                        Text Normalization & Analysis                       */
/* -------------------------------------------------------------------------- */

func normalizeCVText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	sp := regexp.MustCompile(`[ \t]+`)
	s = sp.ReplaceAllString(s, " ")
	bl := regexp.MustCompile(`\n{2,}`)
	s = bl.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}
