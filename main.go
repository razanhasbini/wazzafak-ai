package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	// for PDF text extraction
	"github.com/gorilla/websocket"

	_ "github.com/jackc/pgx/v5/stdlib" // pgx as database/sql driver
	"github.com/joho/godotenv"
	openai "github.com/sashabaranov/go-openai"
)

/* -------------------- app config -------------------- */

type App struct {
	DB        *sql.DB
	LLM       *openai.Client // nil when in MOCK mode
	EmbModel  string
	ChatModel string
	EvalModel string
	Port      string
	Mock      bool
}

func mustEnv(k string) string {
	v := os.Getenv(k)
	if v == "" {
		log.Fatalf("%s is required", k)
	}
	return v
}
func getEnv(k, d string) string {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	return v
}

/* -------------------- bootstrap -------------------- */

func main() {
	_ = godotenv.Load()
	// Required
	dsn := mustEnv("DATABASE_URL")

	// Mock mode lets you run without OpenAI quota
	mock := os.Getenv("USE_MOCK") == "1"
	key := os.Getenv("LLM_API_KEY")
	if !mock && key == "" {
		log.Fatal("LLM_API_KEY is required unless USE_MOCK=1")
	}

	// Config
	port := getEnv("PORT", "8089")
	emb := getEnv("EMBEDDING_MODEL", "text-embedding-3-small")
	chat := getEnv("CHAT_MODEL", "gpt-4o-mini")
	eval := getEnv("EVAL_MODEL", "gpt-4o")

	// DB
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		log.Fatal(err)
	}
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(30 * time.Minute)
	if err := db.Ping(); err != nil {
		log.Fatal(err)
	}
	//here
	// ✅ Test which PostgreSQL instance we're connected to
	var version string
	if err := db.QueryRow("SELECT version()").Scan(&version); err != nil {
		log.Fatalf("❌ Database test query failed: %v", err)
	}
	log.Printf("✅ Connected to database: %s", version)

	// LLM client (only in real mode)
	var llm *openai.Client
	if !mock {
		cfg := openai.DefaultConfig(key)
		if base := os.Getenv("LLM_BASE_URL"); base != "" {
			cfg.BaseURL = base // e.g., Azure/OpenRouter
		}
		llm = openai.NewClientWithConfig(cfg)
	} else {
		log.Println("** MOCK MODE enabled: no calls to OpenAI **")
	}

	app := &App{
		DB:        db,
		LLM:       llm,
		EmbModel:  emb,
		ChatModel: chat,
		EvalModel: eval,
		Port:      port,
		Mock:      mock,
	}

	// HTTP routes
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok":true}`))
	})
	mux.HandleFunc("/v1/kb/seed", app.handleSeedKB)
	mux.HandleFunc("/v1/cv/ingest", app.handleIngestCV)
	mux.HandleFunc("/v1/next-question", app.handleNextQ)
	mux.HandleFunc("/v1/evaluate", app.handleEvaluate)
	mux.HandleFunc("/v1/upload-cv", app.handleUploadCV)
	mux.HandleFunc("/v1/cv/upload", app.handleUploadCV)

	mux.HandleFunc("/v1/live-interview", app.handleLiveInterview)

	// --- Add CORS support ---
	corsHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		mux.ServeHTTP(w, r)
	})

	// --- Server setup ---
	s := &http.Server{
		Addr:              ":" + port,
		Handler:           logReq(corsHandler),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("Interview-AI on :%s", port)
	log.Fatal(s.ListenAndServe())
}

func logReq(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		h.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

/* -------------------- embedding + math -------------------- */

func (a *App) embed(ctx context.Context, texts []string) ([][]float32, error) {
	// Mock mode: deterministic, local vectors (no network)
	if a.LLM == nil {
		out := make([][]float32, len(texts))
		for i, t := range texts {
			out[i] = str2vec(t)
		}
		return out, nil
	}

	// Real mode: call OpenAI embeddings
	resp, err := a.LLM.CreateEmbeddings(ctx, openai.EmbeddingRequestStrings{
		Model: openai.EmbeddingModel(a.EmbModel),
		Input: texts,
	})
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

func str2vec(s string) []float32 {
	d := 256
	v := make([]float32, d)
	var h uint64 = 1469598103934665603
	for _, r := range s {
		h ^= uint64(r)
		h *= 1099511628211
		i := int(h % uint64(d))
		v[i] += 1
	}
	// L2 normalize
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	if sum > 0 {
		n := float32(1.0 / math.Sqrt(sum))
		for i := range v {
			v[i] *= n
		}
	}
	return v
}

func cosSim(a, b []float32) float64 {
	var dot, na, nb float64
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	den := math.Sqrt(na) * math.Sqrt(nb)
	if den == 0 {
		return 0
	}
	return dot / den
}

type Item struct {
	Text string
	Emb  []float32
}

func topKByCos(items []Item, q []float32, k int) []Item {
	type scored struct {
		it Item
		s  float64
	}
	scores := make([]scored, 0, len(items))
	for _, it := range items {
		scores = append(scores, scored{it: it, s: cosSim(it.Emb, q)})
	}
	// simple partial sort
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].s > scores[i].s {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]Item, k)
	for i := 0; i < k; i++ {
		out[i] = scores[i].it
	}
	return out
}

func toJSON(v []float32) []byte {
	b, _ := json.Marshal(v)
	return b
}
func parseEmb(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	var f []float32
	if err := json.Unmarshal(b, &f); err != nil {
		return nil
	}
	return f
}

func chunkText(s string, size, overlap int) []string {
	if size <= overlap {
		overlap = 0
	}
	var out []string
	for i := 0; i < len(s); i += size - overlap {
		end := i + size
		if end > len(s) {
			end = len(s)
		}
		out = append(out, s[i:end])
		if end == len(s) {
			break
		}
	}
	return out
}
func join(snips []string, max int) string {
	if len(snips) > max {
		snips = snips[:max]
	}
	return strings.Join(snips, "\n---\n")
}
func extractJSON(s string) string {
	i := strings.Index(s, "{")
	j := strings.LastIndex(s, "}")
	if i >= 0 && j > i {
		return s[i : j+1]
	}
	b, _ := json.Marshal(s)
	return `{"error":"model did not return JSON","raw":` + string(b) + `}`
}

/* -------------------- HTTP handlers -------------------- */

type seedReq struct {
	Domain string   `json:"domain"`
	Items  []string `json:"items"`
}

func (a *App) handleSeedKB(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var in seedReq
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(in.Domain) == "" || len(in.Items) == 0 {
		http.Error(w, "domain+items required", http.StatusBadRequest)
		return
	}

	embs, err := a.embed(r.Context(), in.Items)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	tx, err := a.DB.Begin()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`INSERT INTO ai.kb_articles(domain,title,text,embedding_json) VALUES ($1,$2,$3,$4::jsonb)`)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer stmt.Close()

	for i, t := range in.Items {
		if _, err := stmt.Exec(in.Domain, fmt.Sprintf("seed-%d", i+1), t, string(toJSON(embs[i]))); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	if err := tx.Commit(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(fmt.Sprintf(`{"ok":true,"inserted":%d}`, len(in.Items))))
}

type ingestReq struct {
	UserID int64  `json:"user_id"`
	Text   string `json:"text"`
}

func (a *App) handleIngestCV(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var in ingestReq
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if in.UserID <= 0 || strings.TrimSpace(in.Text) == "" {
		http.Error(w, "user_id+text required", http.StatusBadRequest)
		return
	}

	tx, err := a.DB.Begin()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer tx.Rollback()

	var cvID int64
	if err := tx.QueryRow(`INSERT INTO ai.cv_documents(user_id,text) VALUES ($1,$2) RETURNING id`, in.UserID, in.Text).Scan(&cvID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	chunks := chunkText(in.Text, 4000, 400)
	embs, err := a.embed(r.Context(), chunks)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	stmt, err := tx.Prepare(`INSERT INTO ai.cv_chunks(cv_id,chunk_text,embedding_json,ord) VALUES ($1,$2,$3::jsonb,$4)`)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer stmt.Close()

	for i := range chunks {
		if _, err := stmt.Exec(cvID, chunks[i], string(toJSON(embs[i])), i); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	if err := tx.Commit(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(fmt.Sprintf(`{"ok":true,"cv_id":%d,"chunks":%d}`, cvID, len(chunks))))
}

type nextQReq struct {
	UserID     int64  `json:"user_id"`
	Domain     string `json:"domain"`
	Difficulty string `json:"difficulty"`
}
type nextQResp struct {
	Question string `json:"question"`
}

func (a *App) handleNextQ(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var in nextQReq
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if in.UserID <= 0 || in.Domain == "" {
		http.Error(w, "user_id+domain required", http.StatusBadRequest)
		return
	}

	// embed the "query"
	q := "next interview question for " + in.Domain
	qEmb, err := a.embed(r.Context(), []string{q})
	if err != nil || len(qEmb) == 0 {
		http.Error(w, "embedding error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	queryEmb := qEmb[0]

	// fetch candidates from DB
	cvRows, err := a.DB.Query(`
	  SELECT c.chunk_text, c.embedding_json
	  FROM ai.cv_chunks c
	  JOIN ai.cv_documents d ON d.id = c.cv_id
	  WHERE d.user_id = $1
	  ORDER BY c.ord
	  LIMIT 200`, in.UserID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer cvRows.Close()

	var cvItems []Item
	for cvRows.Next() {
		var txt string
		var embBytes []byte
		if err := cvRows.Scan(&txt, &embBytes); err == nil {
			cvItems = append(cvItems, Item{Text: txt, Emb: parseEmb(embBytes)})
		}
	}

	kbRows, err := a.DB.Query(`
	  SELECT text, embedding_json
	  FROM ai.kb_articles
	  WHERE domain = $1
	  LIMIT 200`, in.Domain)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer kbRows.Close()

	var kbItems []Item
	for kbRows.Next() {
		var txt string
		var embBytes []byte
		if err := kbRows.Scan(&txt, &embBytes); err == nil {
			kbItems = append(kbItems, Item{Text: txt, Emb: parseEmb(embBytes)})
		}
	}

	ctxItems := append(topKByCos(cvItems, queryEmb, 3), topKByCos(kbItems, queryEmb, 3)...)
	var snips []string
	for _, it := range ctxItems {
		snips = append(snips, it.Text)
	}
	contextBlock := join(snips, 6)

	// MOCK: return a canned question
	if a.LLM == nil {
		q := "Based on your background, describe how you would design a simple REST API in " + in.Domain + "."
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(nextQResp{Question: q})
		return
	}

	// REAL: ask the model
	sys := fmt.Sprintf(`You are a strict but fair interviewer in the %s domain.
Ask ONE question (<=2 sentences). Use CONTEXT if needed. Do not reveal answers.`, in.Domain)
	user := "CONTEXT:\n" + contextBlock + "\n---\nDifficulty: " + in.Difficulty + "\nReturn exactly one interview question now."

	resp, err := a.LLM.CreateChatCompletion(r.Context(), openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.3,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	question := strings.TrimSpace(resp.Choices[0].Message.Content)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(nextQResp{Question: question})
}

type evalReq struct {
	SessionID string `json:"session_id"`
	UserID    int64  `json:"user_id"`
	Domain    string `json:"domain"`
	Question  string `json:"question"`
	Answer    string `json:"answer"`
}

func (a *App) handleEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var in evalReq
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if in.UserID <= 0 || in.Domain == "" || in.Question == "" || in.Answer == "" {
		http.Error(w, "user_id,domain,question,answer required", http.StatusBadRequest)
		return
	}

	qaEmb, err := a.embed(r.Context(), []string{in.Question + " " + in.Answer})
	if err != nil || len(qaEmb) == 0 {
		http.Error(w, "embedding error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	queryEmb := qaEmb[0]

	kbRows, err := a.DB.Query(`
	  SELECT text, embedding_json
	  FROM ai.kb_articles
	  WHERE domain = $1
	  LIMIT 200`, in.Domain)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer kbRows.Close()

	var kbItems []Item
	for kbRows.Next() {
		var txt string
		var embBytes []byte
		if err := kbRows.Scan(&txt, &embBytes); err == nil {
			kbItems = append(kbItems, Item{Text: txt, Emb: parseEmb(embBytes)})
		}
	}
	ctxItems := topKByCos(kbItems, queryEmb, 6)
	var snips []string
	for _, it := range ctxItems {
		snips = append(snips, it.Text)
	}
	ctxBlock := join(snips, 6)

	// MOCK: return a canned JSON result
	if a.LLM == nil {
		mock := `{
		  "scores":[
		    {"name":"Relevance","score":4},
		    {"name":"Structure/STAR","score":3},
		    {"name":"Technical Correctness","score":4},
		    {"name":"Clarity & Conciseness","score":3},
		    {"name":"Confidence & Tone","score":4}
		  ],
		  "overall":78,
		  "strengths":["Good coverage of key concepts","Clear tech stack"],
		  "weaknesses":["Missing concrete metrics","Shallow on edge cases"],
		  "action_items":["Add latency/error budgets","Describe pagination and auth flows"],
		  "suggested_model_answer":"Define endpoints and resources, use proper HTTP methods/status codes, validate inputs, context timeouts, structured logs, and defensive DB access."
		}`
		_, _ = a.DB.Exec(
			`INSERT INTO ai.evaluations(session_id,question,answer,result_json) VALUES ($1,$2,$3,$4::jsonb)`,
			in.SessionID, in.Question, in.Answer, mock,
		)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(mock))
		return
	}

	// REAL: call the model to evaluate
	sys := "You are an interview evaluator. Output STRICT JSON only, no prose."
	user := fmt.Sprintf(`QUESTION: %s
ANSWER: %s
CONTEXT:
%s

RUBRIC:
{
  "criteria": [
    {"name":"Relevance","weight":0.25},
    {"name":"Structure/STAR","weight":0.20},
    {"name":"Technical Correctness","weight":0.30},
    {"name":"Clarity & Conciseness","weight":0.15},
    {"name":"Confidence & Tone","weight":0.10}
  ],
  "scale":{"min":1,"max":5}
}

Return JSON:
{
  "scores":[{"name":"Relevance","score":1-5},{"name":"Structure/STAR","score":1-5},{"name":"Technical Correctness","score":1-5},{"name":"Clarity & Conciseness","score":1-5},{"name":"Confidence & Tone","score":1-5}],
  "overall":1-100,
  "strengths":["..."],
  "weaknesses":["..."],
  "action_items":["..."],
  "suggested_model_answer":"..."
}`, in.Question, in.Answer, ctxBlock)

	resp, err := a.LLM.CreateChatCompletion(r.Context(), openai.ChatCompletionRequest{
		Model:       a.EvalModel,
		Temperature: 0.0,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	jsonText := extractJSON(raw)

	_, _ = a.DB.Exec(
		`INSERT INTO ai.evaluations(session_id,question,answer,result_json) VALUES ($1,$2,$3,$4::jsonb)`,
		in.SessionID, in.Question, in.Answer, jsonText,
	)

	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(jsonText))
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func (a *App) handleLiveInterview(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	defer conn.Close()

	log.Println("🎙️ New live interview session started")

	// Extract user_id from query
	query := r.URL.Query()
	userIDStr := query.Get("user_id")
	var userID int64
	if userIDStr != "" {
		fmt.Sscan(userIDStr, &userID)
	}

	// 🧠 Load CV content for context (if available)
	var cvText string
	if userID > 0 {
		err := a.DB.QueryRow(`SELECT text FROM ai.cv_documents WHERE user_id = $1 ORDER BY id DESC LIMIT 1`, userID).Scan(&cvText)
		if err != nil && err != sql.ErrNoRows {
			log.Println("Error loading CV:", err)
		}
	}

	// 🧠 Construct a rich, adaptive system prompt
	systemPrompt := fmt.Sprintf(`You are a professional interviewer conducting a realistic mock interview.
Analyze the candidate’s background from their CV and tailor the interview accordingly.

Candidate CV (if available):
%s

Rules:
1. Begin immediately by greeting the candidate and introducing yourself naturally (e.g. "Hi, nice to meet you, I’ll be interviewing you today.").
2. Based on the CV, identify the candidate’s likely field — for example:
   - Software Engineering → ask about programming, databases, or systems design.
   - Marketing → ask about strategy, analytics, or campaigns.
   - Business → ask about management, negotiation, or finance.
   - Mechanical Engineering → ask about design, materials, or manufacturing.
   - Medicine → ask about diagnosis, teamwork, or patient care.
3. Ask **one** question at a time.
4. When the candidate answers, analyze their response and ask a deeper follow-up question.
5. If answers are vague, focus on one specific element ("Can you elaborate on how you approached that project?").
6. Occasionally test soft skills (communication, problem solving, teamwork).
7. After about 10–12 turns, thank the candidate and conclude politely.
8. Maintain a natural, friendly, and professional tone throughout.
9. Do not provide answers — only ask or respond conversationally like a real interviewer.`, cvText)

	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
	}

	// 🗣️ Immediately generate the AI’s greeting / first question
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	resp, err := a.LLM.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.7,
		Messages:    messages,
	})
	cancel()

	if err != nil {
		log.Println("Error starting interview:", err)
		conn.WriteMessage(websocket.TextMessage, []byte("Error starting interview: "+err.Error()))
		return
	}

	firstReply := strings.TrimSpace(resp.Choices[0].Message.Content)
	log.Println("🤖 Interviewer:", firstReply)
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: firstReply,
	})
	conn.WriteMessage(websocket.TextMessage, []byte(firstReply))

	//  Conversation loop
	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Println("read error:", err)
			break
		}

		userMsg := string(msg)
		log.Println("👤 Candidate:", userMsg)

		if a.Mock {
			reply := "Interesting — could you tell me a bit more about that?"
			conn.WriteMessage(websocket.TextMessage, []byte(reply))
			continue
		}

		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userMsg,
		})

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		resp, err := a.LLM.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:       a.ChatModel,
			Temperature: 0.7,
			Messages:    messages,
		})
		cancel()

		if err != nil {
			log.Println("AI error:", err)
			conn.WriteMessage(websocket.TextMessage, []byte("Error: "+err.Error()))
			continue
		}

		reply := strings.TrimSpace(resp.Choices[0].Message.Content)
		log.Println("🤖 AI:", reply)
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: reply,
		})
		conn.WriteMessage(websocket.TextMessage, []byte(reply))
	}

	log.Println(" Live interview ended")
}
