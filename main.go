package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/joho/godotenv"
	openai "github.com/sashabaranov/go-openai"
)

/* -------------------- app config -------------------- */

// Global OpenAI client
var client *openai.Client

func initOpenAIClient() {
	var err error
	apiKey := mustEnv("LLM_API_KEY")  // Fetch the API key from environment variables
	client = openai.NewClient(apiKey) // Initialize the OpenAI client
	if err != nil {
		log.Fatalf("Error initializing OpenAI client: %v", err)
	}
}

// Function to get stress level from Python service
func getStressLevel(userID int64) (string, error) {
	// Send request to Python microservice for stress level
	resp, err := http.Post(fmt.Sprintf("http://localhost:5001/analyze_stress"),
		"application/json", nil) // Add actual data if needed
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result["stress_level"].(string), nil
}

type App struct {
	DB        *sql.DB
	LLM       *openai.Client // nil when in MOCK mode
	EmbModel  string
	ChatModel string
	EvalModel string
	Port      string
	Mock      bool
	Dataset   *DatasetManager
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
	initOpenAIClient()
	dsn := mustEnv("DATABASE_URL")

	mock := os.Getenv("USE_MOCK") == "1"
	key := os.Getenv("LLM_API_KEY")
	if !mock && key == "" {
		log.Fatal("LLM_API_KEY is required unless USE_MOCK=1")
	}

	port := getEnv("PORT", "8089")
	emb := getEnv("EMBEDDING_MODEL", "text-embedding-3-small")
	chat := getEnv("CHAT_MODEL", "gpt-4o-mini") // swap later with your fine-tuned model id
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
	var version string
	if err := db.QueryRow("SELECT version()").Scan(&version); err != nil {
		log.Fatalf("❌ Database test query failed: %v", err)
	}
	log.Printf("✅ Connected to database: %s", version)

	// LLM
	var llm *openai.Client
	if !mock {
		cfg := openai.DefaultConfig(key)
		if base := os.Getenv("LLM_BASE_URL"); base != "" {
			cfg.BaseURL = base
		}
		llm = openai.NewClientWithConfig(cfg)
	} else {
		log.Println("** MOCK MODE enabled: no calls to OpenAI **")
	}

	// Load interview datasets (JSON)
	dm, err := LoadDatasets()
	if err != nil {
		log.Println("❌ Failed to load interview datasets:", err)
		// continue without datasets (the AI will still work, just less guided)
	}

	app := &App{
		DB:        db,
		LLM:       llm,
		EmbModel:  emb,
		ChatModel: chat,
		EvalModel: eval,
		Port:      port,
		Mock:      mock,
		Dataset:   dm,
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
	// CV upload endpoint implemented in cv_parser.go
	mux.HandleFunc("/v1/upload-cv", app.handleUploadCV)
	mux.HandleFunc("/v1/cv/upload", app.handleUploadCV)

	mux.HandleFunc("/v1/live-interview", app.handleLiveInterview)

	// CORS
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
	if a.LLM == nil {
		out := make([][]float32, len(texts))
		for i, t := range texts {
			out[i] = str2vec(t)
		}
		return out, nil
	}
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

/* -------------------- HTTP handlers already in your app -------------------- */

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

/* -------------------- Interview engine (WebSocket) -------------------- */

/* -------------------- Interview engine (WebSocket) -------------------- */

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Tracks the live interview state for one candidate
type InterviewState struct {
	Turn              int
	Responses         []string
	TechAsked         int
	Field             string
	Subfield          string
	Skills            []string
	Projects          []string
	Experiences       []string
	LastUser          string
	LastQuestion      string
	Asked             []string
	InCandidateQPhase bool
}

// Pick one relevant project depending on question type
func pickRelevantProject(st *InterviewState, qType string) string {
	if len(st.Projects) == 0 {
		return ""
	}

	// For technical questions: pick one that sounds "techy"
	if qType == "technical" {
		for _, p := range st.Projects {
			lp := strings.ToLower(p)
			if strings.Contains(lp, "api") ||
				strings.Contains(lp, "backend") ||
				strings.Contains(lp, "integration") ||
				strings.Contains(lp, "real-time") {
				return p
			}
		}
	}

	// Otherwise: return the longest project text
	longest := ""
	for _, p := range st.Projects {
		if len(p) > len(longest) {
			longest = p
		}
	}
	return longest
}

func alreadyAsked(st *InterviewState, q string) bool {
	q = strings.TrimSpace(strings.ToLower(q))
	for _, prev := range st.Asked {
		if strings.TrimSpace(strings.ToLower(prev)) == q {
			return true
		}
	}
	return false
}
func tooSimilar(newQ string, prevQs []string) bool {
	newQ = strings.ToLower(newQ)

	for _, old := range prevQs {
		old = strings.ToLower(old)

		// 1) same starting verb
		if strings.HasPrefix(newQ, strings.Split(old, " ")[0]) {
			return true
		}

		// 2) repeated structure
		if strings.Contains(newQ, "explain") && strings.Contains(old, "explain") {
			return true
		}
		if strings.Contains(newQ, "describe") && strings.Contains(old, "describe") {
			return true
		}
		if strings.Contains(newQ, "how did you") && strings.Contains(old, "how did you") {
			return true
		}

		// 3) project-based repetition
		if strings.Contains(newQ, "project") && strings.Contains(old, "project") {
			return true
		}

		// 4) overlap of meaningful words
		wordsNew := strings.Fields(newQ)
		wordsOld := strings.Fields(old)
		matches := 0

		for _, wn := range wordsNew {
			if len(wn) <= 4 {
				continue
			}
			for _, wo := range wordsOld {
				if wn == wo {
					matches++
					if matches >= 4 {
						return true
					}
				}
			}
		}
	}
	return false
}

// Dataset question helper – NEVER used for intro / motivation
func (a *App) datasetQuestionFor(st *InterviewState, qType string) string {
	if a.Dataset == nil {
		return ""
	}
	// Do not use dataset questions for the first two turns
	if qType == "personal_intro" || qType == "motivation" {
		return ""
	}

	lines := a.datasetFor(st.Field, st.Subfield)
	if len(lines) == 0 {
		return ""
	}

	var candidates []string
	for _, line := range lines {
		if !alreadyAsked(st, line) {
			candidates = append(candidates, line)
		}
	}
	if len(candidates) == 0 {
		return ""
	}
	return candidates[rand.Intn(len(candidates))]
}

// Function to generate feedback based on the responses
func generateInterviewFeedback(responses []string) (string, error) {
	// Prepare a prompt for OpenAI to generate feedback
	feedbackPrompt := "Based on the following interview responses, provide feedback on the candidate's performance. Evaluate the answers for confidence, clarity, and technical correctness. Provide suggestions for improvement if needed.\n\n"

	// Append all responses to the prompt
	for i, response := range responses {
		feedbackPrompt += fmt.Sprintf("Response %d: %s\n", i+1, response)
	}

	// Call OpenAI to generate feedback
	openAIResponse, err := client.CreateCompletion(context.Background(), openai.CompletionRequest{
		Model:     openai.GPT3TextDavinci003, // Specify the model you are using
		Prompt:    feedbackPrompt,
		MaxTokens: 500, // Increase tokens if feedback needs more detail
	})

	if err != nil {
		return "", err
	}

	// Extract the generated feedback text from OpenAI's response
	feedback := openAIResponse.Choices[0].Text
	return feedback, nil
}

// Function to store the response feedback, sentiment, and confidence in the feedback table
func (a *App) storeInterviewFeedback(interviewID int64, responseFeedback, sentiment, confidence string) error {
	// Insert the feedback data into the feedback table
	_, err := a.DB.Exec(`
        INSERT INTO public.feedback (interview_id, response_feedback, sentiment, confidence, created_at)
        VALUES ($1, $2, $3, $4, NOW())`, interviewID, responseFeedback, sentiment, confidence)
	return err
}

// Function to analyze the user's response using OpenAI GPT model
// Function to analyze the user's response using OpenAI GPT model
func analyzeAnswerUsingOpenAI(answer string) (string, string, string) {
	// Construct the prompt properly
	prompt := fmt.Sprintf(`
    Analyze the following interview answer for clarity, confidence, tone, and sentiment.
    Provide feedback on how the answer can be improved, and mention if the answer seems confident, nervous, or unsure.
    Also, analyze the sentiment of the answer. Here is the answer:
    "%s"`, answer)

	// Call OpenAI to analyze the answer
	resp, err := client.CreateCompletion(context.Background(), openai.CompletionRequest{
		Prompt:      prompt, // Pass the constructed prompt here
		MaxTokens:   200,    // You can adjust this depending on how long the response should be
		Temperature: 0.7,    // Adjust temperature for creativity
	})
	if err != nil {
		log.Println("Error analyzing the answer:", err)
		return "Error in analysis", "", ""
	}

	// Accessing the response from OpenAI
	analysis := resp.Choices[0].Text // Use 'resp' instead of 'res'
	sentiment := "Neutral"           // Default value for sentiment

	// Simple sentiment classification (this can be enhanced)
	if strings.Contains(analysis, "positive") {
		sentiment = "Positive"
	} else if strings.Contains(analysis, "negative") {
		sentiment = "Negative"
	}

	// Return feedback along with sentiment and confidence
	return analysis, sentiment, "Confidence level: " + sentiment
}

// After the interview ends, generate and send feedback
func endInterviewAndSendFeedback(conn *websocket.Conn, st *InterviewState) {
	feedback, err := generateInterviewFeedback(st.Responses)
	if err != nil {
		log.Println("Error generating feedback:", err)
		return
	}

	err = conn.WriteMessage(websocket.TextMessage, []byte(feedback))
	if err != nil {
		log.Println("Error sending feedback:", err)
	}

	log.Println("Interview ended. Feedback sent to client.")
}

func (a *App) handleLiveInterview(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("WebSocket upgrade error:", err)
		return
	}
	defer conn.Close()

	log.Println("🎙️ New live interview session started")

	// Extract user_id and interview_id from query
	query := r.URL.Query()
	userIDStr := query.Get("user_id")
	interviewIDStr := query.Get("interview_id") // Getting interviewID
	var userID, interviewID int64
	if userIDStr != "" {
		fmt.Sscan(userIDStr, &userID)
	}
	if interviewIDStr != "" {
		fmt.Sscan(interviewIDStr, &interviewID)
	}

	// Load latest CV text for this user
	var cvText string
	if userID > 0 {
		err := a.DB.QueryRow(`SELECT text FROM ai.cv_documents WHERE user_id = $1 ORDER BY id DESC LIMIT 1`, userID).Scan(&cvText)
		if err != nil && err != sql.ErrNoRows {
			log.Println("Error loading CV:", err)
		}
	}

	// Analyze CV structure once
	st := &InterviewState{Turn: 0, TechAsked: 0}
	if strings.TrimSpace(cvText) != "" && a.LLM != nil {
		if fld, sub, skills, projs, exps := a.extractStructure(r.Context(), cvText); fld != "" {
			st.Field = fld
			st.Subfield = sub
			st.Skills = skills
			st.Projects = projs
			st.Experiences = exps
		}
	}

	// 1) Send friendly intro ONLY (no interview question yet)
	opening := a.firstQuestion(r.Context(), st, cvText)
	st.LastQuestion = opening
	_ = conn.WriteMessage(websocket.TextMessage, []byte(opening))

	// Conversation loop
	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Println("read error:", err)
			break
		}
		userMsg := strings.TrimSpace(string(msg))
		if userMsg == "" {
			continue
		}
		st.Responses = append(st.Responses, userMsg) // <-- This line stores the user response

		st.LastUser = userMsg

		// 3) Analyze the last user response using OpenAI
		if st.Turn >= 9 {  // Or whatever your condition is for the end of the interview
			// Only generate feedback after the interview ends
			endInterviewAndSendFeedback(conn, st)
			break  // End the interview here
		}
		


		if err != nil {
			log.Println("Error sending feedback message:", err)
			return
		}

		// 2) Confusion handling: rephrase the last question
		if isConfused(userMsg) && st.LastQuestion != "" {
			reply := a.rephraseQuestion(r.Context(), st)
			st.LastQuestion = reply
			st.Asked = append(st.Asked, reply)
			_ = conn.WriteMessage(websocket.TextMessage, []byte(reply))
			continue
		}

		// 3) Candidate question phase (end of interview)
		if st.InCandidateQPhase {
			reply := a.answerCandidateQuestion(r.Context(), st, userMsg)
			st.LastQuestion = reply
			st.Asked = append(st.Asked, reply)
			_ = conn.WriteMessage(websocket.TextMessage, []byte(reply))

			// Soft or neutral closing depending on emotion
			var final string
			if isEmotionalEnding(st.LastUser) {
				final = "Thank you for sharing your thoughts today. You expressed yourself well, and we appreciate the effort you put into this interview."
			} else {
				final = a.generateClosingMessage(r.Context(), st)
			}

			// Analyze the last user response using OpenAI
			responseFeedback, sentiment, confidence := analyzeAnswerUsingOpenAI(st.LastUser)

			// Store the feedback, sentiment, and confidence in the feedback table
			err = a.storeInterviewFeedback(interviewID, responseFeedback, sentiment, confidence)
			if err != nil {
				log.Println("Error storing feedback:", err)
			}

			final += fmt.Sprintf("\nStress Level: %s\nSentiment: %s\nConfidence: %s", "neutral", sentiment, confidence) // Example for stress level
			_ = conn.WriteMessage(websocket.TextMessage, []byte(final))
			break
		}

		// Generate and send feedback
		endInterviewAndSendFeedback(conn, st)

		// 4) Optional follow-up (rare, only on long answers)
		if st.Turn > 0 && len(userMsg) > 40 && !isConfused(userMsg) {
			if follow := a.maybeFollowUp(r.Context(), st); follow != "" {
				st.LastQuestion = follow
				st.Asked = append(st.Asked, follow)
				_ = conn.WriteMessage(websocket.TextMessage, []byte(follow))
				continue
			}
		}

		// 5) Advance turn and possibly switch to closing
		st.Turn++ // Turn 1 = "Tell me about yourself", Turn 2 = motivation, etc.

		// After ~9 questions, move to candidate question phase
		if st.Turn >= 9 {
			ack := a.generateAcknowledgement(r.Context(), st)
			closing := strings.TrimSpace(ack + " Before we wrap up, do you have any questions for us about the role or the company?")
			st.LastQuestion = closing
			st.Asked = append(st.Asked, closing)
			st.InCandidateQPhase = true
			_ = conn.WriteMessage(websocket.TextMessage, []byte(closing))
			continue
		}

		// 6) Generate main interview question for this turn
		question := a.nextInterviewQuestion(r.Context(), st, cvText)
		if question == "" {
			question = "Tell me about a project from your CV that you're proud of and why."
		}

		// 7) Generate acknowledgement (only if answer was meaningful)
		ack := ""
		if len(userMsg) > 20 {
			ack = a.generateAcknowledgement(r.Context(), st)
		}

		msgOut := strings.TrimSpace(strings.TrimSpace(ack) + " " + strings.TrimSpace(question))

		st.LastQuestion = question
		st.Asked = append(st.Asked, question)

		_ = conn.WriteMessage(websocket.TextMessage, []byte(msgOut))
	}

	log.Println("🔴 Live interview ended")
}

/* -------------------- Interview brain -------------------- */

// Map CV field/subfield to dataset key and sample domain questions
func (a *App) datasetFor(field, sub string) []string {
	if a.Dataset == nil {
		return nil
	}

	f := strings.ToLower(field)
	s := strings.ToLower(sub)
	key := ""

	// SOFTWARE
	if strings.Contains(f, "software") {
		if strings.Contains(s, "front") {
			key = "software_frontend"
		} else if strings.Contains(s, "back") {
			key = "software_backend"
		} else if strings.Contains(s, "full") {
			key = "software_fullstack"
		} else if strings.Contains(s, "mobile") {
			key = "software_mobile"
		}
	}

	// BUSINESS
	if strings.Contains(f, "business") {
		if strings.Contains(s, "finance") {
			key = "business_finance"
		} else if strings.Contains(s, "hr") {
			key = "business_hr"
		} else if strings.Contains(s, "manage") {
			key = "business_management"
		} else if strings.Contains(s, "market") {
			key = "business_marketing"
		} else if strings.Contains(s, "operation") {
			key = "business_operations"
		}
	}

	// MECHANICAL
	if strings.Contains(f, "mechanical") {
		if strings.Contains(s, "aero") {
			key = "mechanical_aerospace"
		} else if strings.Contains(s, "auto") {
			key = "mechanical_automotive"
		} else if strings.Contains(s, "manufact") {
			key = "mechanical_manufacturing"
		}
	}

	// DESIGN
	if strings.Contains(f, "design") || strings.Contains(f, "graphic") {
		if strings.Contains(s, "ui") || strings.Contains(s, "ux") {
			key = "design_uiux"
		} else if strings.Contains(s, "brand") {
			key = "design_branding"
		} else if strings.Contains(s, "illustr") {
			key = "design_illustration"
		}
	}

	// DIGITAL MARKETING
	if strings.Contains(f, "marketing") {
		if strings.Contains(s, "digital") {
			key = "marketing_digital"
		} else if strings.Contains(s, "content") {
			key = "marketing_content"
		} else if strings.Contains(s, "seo") {
			key = "marketing_seo"
		}
	}

	// Fallback: fuzzy match any dataset key
	if key == "" && a.Dataset != nil {
		for k := range a.Dataset.Data {
			if strings.Contains(k, f) || strings.Contains(k, s) {
				key = k
				break
			}
		}
	}

	if key == "" {
		return nil
	}

	// Mix of technical + scenario + behavioral for style hints
	var lines []string
	lines = append(lines, a.Dataset.SampleQuestions(key, "technical", 2)...)
	lines = append(lines, a.Dataset.SampleQuestions(key, "scenario", 2)...)
	lines = append(lines, a.Dataset.SampleQuestions(key, "behavioral", 2)...)

	if len(lines) == 0 {
		lines = a.Dataset.SampleQuestions(key, "", 4)
	}

	return lines
}

func (a *App) extractStructure(ctx context.Context, cv string) (field, sub string, skills, projects, exps []string) {
	if a.LLM == nil {
		return "", "", nil, nil, nil
	}
	sys := `
You are a precise CV parser.

Task:
Extract structured information from the CV and return STRICT JSON.

Rules:
- Field must be one of:
  "Software Engineering", "Business", "Mechanical Engineering",
  "Graphic Design", "Marketing", "Other"
- Subfield must be a short phrase (e.g., Frontend, Backend, Finance, UI/UX, etc.)
- Skills: list of individual skills (not sentences)
- Projects: 1–2 sentence summaries or titles
- Experiences: short role@company descriptions

No commentary outside JSON.
`
	user := fmt.Sprintf(`Extract structured info from this CV.

CV:
---
%s
---

Return JSON:
{
  "field": "Software Engineering | Business | Mechanical Engineering | Graphic Design | Marketing | Other",
  "subfield": "short phrase, e.g., Frontend, Backend, Full Stack, Mobile, HR, Finance, Operations, Marketing, Automotive, Aerospace, Manufacturing, UI/UX, Branding, Illustration, Digital Marketing, Content Marketing, SEO, etc.",
  "skills": ["skill1","skill2",...],
  "projects": ["project summary or title", "..."],
  "experiences": ["role@company with brief context", "..."]
}`, cv)

	ctx2, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.1,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return "", "", nil, nil, nil
	}
	js := extractJSON(strings.TrimSpace(resp.Choices[0].Message.Content))
	var out struct {
		Field       string   `json:"field"`
		Subfield    string   `json:"subfield"`
		Skills      []string `json:"skills"`
		Projects    []string `json:"projects"`
		Experiences []string `json:"experiences"`
	}
	_ = json.Unmarshal([]byte(js), &out)
	return strings.TrimSpace(out.Field), strings.TrimSpace(out.Subfield), out.Skills, out.Projects, out.Experiences
}

// First message: greeting + explain structure + ask how they feel
func (a *App) firstQuestion(ctx context.Context, st *InterviewState, cv string) string {
	if a.LLM == nil {
		return "Hi there, I’m your interviewer today here at [Company Name]. We’ll go through some general questions and then some based on your CV. Before we dive in, how are you feeling today?"
	}

	sys := `
You are a professional recruiter conducting a live video interview.

Task:
Generate a natural spoken introduction in 2–3 sentences.

Content rules:
- Sentence 1: Introduce yourself and your role.
- Sentence 2: Briefly explain the interview structure (general questions + CV-based questions).
- Sentence 3: Politely ask how they are feeling before starting.

Tone:
Warm, human, confident, not robotic.

Do NOT:
- Ask any interview or CV questions yet.
- Use emojis or lists.
`
	user := "Generate the spoken introduction now."

	ctx2, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.4,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return "Hi, I’m your interviewer today. We’ll start with a few general questions and then some based on your CV. How are you feeling before we begin?"
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

// Decide question type strictly by turn number
func decideType(st *InterviewState) string {
	switch st.Turn {
	case 1:
		return "personal_intro" // Tell me about yourself
	case 2:
		return "motivation" // Why this role / area
	case 3, 4, 5:
		return "technical" // CV-based technical
	case 6:
		return "scenario" // Work scenario
	case 7:
		return "experience" // Deep-dive project/experience
	case 8:
		return "behavioral" // conflict/teamwork/ownership
	default:
		if st.Turn%2 == 0 {
			return "experience"
		}
		return "scenario"
	}
}

// Main question generator
func (a *App) nextInterviewQuestion(ctx context.Context, st *InterviewState, cv string) string {
	qType := decideType(st)
	var domainLines []string

	// 1) Fully deterministic for the first two questions
	if qType == "personal_intro" {
		return "To start off, could you tell me a bit about yourself and your background?"
	}
	if qType == "motivation" {
		return "What motivated you to apply for this type of role and work in this kind of environment?"
	}

	// 2) First preference: CV-tailored question (for tech/experience/scenario/etc.)
	if qType != "personal_intro" && qType != "motivation" {
		if cvQ := a.generateCVTailoredQuestion(ctx, st, qType); cvQ != "" {
			if qType == "technical" {
				st.TechAsked++
			}
			return cvQ
		}
	}

	// 3) Second preference: dataset question (only after intro/motivation)
	if q := a.datasetQuestionFor(st, qType); q != "" {
		if qType == "technical" {
			st.TechAsked++
		}
		return q
	}

	// 4) LLM generic interview question, using CV structure + previous questions as context
	if a.Dataset != nil {
		domainLines = a.datasetFor(st.Field, st.Subfield)
	}

	// Build "previous questions" block
	var prevBlock string
	if len(st.Asked) == 0 {
		prevBlock = "(none)"
	} else {
		var rows []string
		for _, q := range st.Asked {
			q = strings.TrimSpace(q)
			if q == "" {
				continue
			}
			rows = append(rows, "- "+q)
		}
		if len(rows) == 0 {
			prevBlock = "(none)"
		} else {
			prevBlock = strings.Join(rows, "\n")
		}
	}

	typeLine := ""
	switch qType {
	case "experience":
		typeLine = "Ask about a specific project or experience from their CV. Prefer concrete projects if available. Ask for impact, decisions, or challenges."
	case "technical":
		typeLine = "Ask a focused technical question grounded in their listed skills or projects. One concept only, 1–2 sentences."
	case "scenario":
		typeLine = "Ask a realistic work scenario question that fits their field and subfield."
	case "behavioral":
		typeLine = "Ask about past behavior, such as conflict, teamwork, ownership, or dealing with pressure."
	case "hr_why_role":
		typeLine = "Ask why they are interested in this specific role."
	case "hr_why_company":
		typeLine = "Ask why they want to work at this company."
	case "hr_strengths":
		typeLine = "Ask about their greatest strengths relevant to this role."
	case "hr_weaknesses":
		typeLine = "Ask about a genuine weakness they are working on."
	case "hr_salary":
		typeLine = "Ask politely about their salary expectations for this role."
	default:
		typeLine = "Ask a professional interview question linked to their background."
	}

	sys := `
You are a senior recruiter interviewing a candidate in a live conversation.

GOAL:
Generate ONE and ONLY ONE interview question.

GENERAL RULES:
- Maximum 2 sentences.
- Use QUESTION_TYPE_INSTRUCTION to choose the topic.
- Prefer using their CV (skills, projects, experiences) when relevant.
- Sound like a human interviewer, not an AI.
- Vary your language naturally.
- Avoid repeating the same question or wording from PREVIOUS_QUESTIONS.

MUST NOT:
- Do NOT paraphrase or repeat the candidate's previous answers.
- Do NOT ask more than one question.
- Do NOT say "Thanks", "Understood", "Noted", or any acknowledgement.
- Do NOT use emojis.

OUTPUT:
Return only the question text.
`
	user := fmt.Sprintf(`STRUCTURED_CV:
Field=%s | Subfield=%s
Skills=%v
RelevantProject=%s
Experiences=%v

PREVIOUS_QUESTIONS:
%s

DOMAIN_QUESTION_EXAMPLES:
%s

QUESTION_TYPE_INSTRUCTION:
%s
`,
		st.Field, st.Subfield,
		st.Skills,
		pickRelevantProject(st, qType),
		st.Experiences,
		prevBlock,
		sampleLines(domainLines, 3),
		typeLine,
	)

	if a.LLM == nil {
		// Simple fallback if no LLM
		switch qType {
		case "technical":
			st.TechAsked++
			return "Could you describe a technical challenge from one of your projects and how you solved it?"
		case "scenario":
			return "Imagine you join a project that is behind schedule. What would you do first?"
		case "behavioral":
			return "Tell me about a time you had a conflict with someone at work and how you handled it."
		default:
			return "Could you walk me through one of the main projects on your CV and your contribution?"
		}
	}

	ctx2, cancel := context.WithTimeout(ctx, 40*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.5,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		switch qType {
		case "technical":
			st.TechAsked++
			return "Could you describe a technical challenge from one of your projects and how you solved it?"
		case "scenario":
			return "Imagine you join a project that is behind schedule. What would you do first?"
		case "behavioral":
			return "Tell me about a time you had a conflict with someone at work and how you handled it."
		default:
			return "Could you walk me through one of the main projects on your CV and your contribution?"
		}
	}

	out := strings.TrimSpace(resp.Choices[0].Message.Content)
	newQ := out

	// Prevent repeated or similar questions
	if tooSimilar(newQ, lastN(st.Asked, 4)) {

		// fallback #1: try dataset question
		alt := a.datasetQuestionFor(st, qType)
		if alt != "" && !tooSimilar(alt, lastN(st.Asked, 4)) {
			newQ = alt
		} else {
			// fallback #2: regenerate CV question (fresh prompt)
			newQ = a.generateCVTailoredQuestion(ctx, st, qType)
		}
	}
	if qType == "technical" {
		st.TechAsked++
	}
	return newQ
}

func (a *App) answerCandidateQuestion(ctx context.Context, st *InterviewState, question string) string {
	q := strings.TrimSpace(question)
	if q == "" {
		return "Those are great questions, and the hiring manager will share more details with you in the next stage."
	}

	sys := `
You are a recruiter. The candidate is asking a question about the role or company.
RULES:
- Answer in 2–3 sentences.
- Be concise and realistic.
- If you don't know a detail, give a professional generic answer.
- No emojis. No bullet points.
`
	user := fmt.Sprintf("Candidate question: %q\nField: %s | Subfield: %s", q, st.Field, st.Subfield)

	ctx2, cancel := context.WithTimeout(ctx, 25*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.4,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return "Great question — the hiring team will share more specifics with you in the next interview stage."
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

/* -------------------- Helper functions for live interview -------------------- */

func isConfused(msg string) bool {
	m := strings.ToLower(msg)
	confusionIndicators := []string{
		"don't understand", "dont understand", "don’t understand",
		"i'm confused", "im confused",
		"not sure what you mean",
		"what do you mean",
		"can you rephrase",
		"can you say it differently",
		"you are repeating",
		"this is repeated",
		"same question",
	}
	for _, c := range confusionIndicators {
		if strings.Contains(m, c) {
			return true
		}
	}
	return false
}

func isEmotional(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "anxious") ||
		strings.Contains(m, "anxiety") ||
		strings.Contains(m, "nervous") ||
		strings.Contains(m, "stressed") ||
		strings.Contains(m, "worried") ||
		strings.Contains(m, "sad") ||
		strings.Contains(m, "frustrated") ||
		strings.Contains(m, "overwhelmed") ||
		strings.Contains(m, "not confident") ||
		strings.Contains(m, "i'm scared") ||
		strings.Contains(m, "im scared") ||
		strings.Contains(m, "i feel bad")
}

var softEmpathyAcks = []string{
	"I understand how that could feel stressful.",
	"I hear you, and that’s completely valid.",
	"I get that this can feel a bit overwhelming.",
}

func (a *App) rephraseQuestion(ctx context.Context, st *InterviewState) string {
	orig := strings.TrimSpace(st.LastQuestion)
	if orig == "" {
		return "Let me phrase that differently: could you walk me through your thought process in that situation?"
	}

	if a.LLM == nil {
		return "Let me say it a different way: " + orig
	}

	sys := `
You are a recruiter rephrasing a question because the candidate is confused.

RULES:
- Change the angle completely (not the wording only).
- Make it easier, simpler, more concrete.
- Remove technical wording.
- ONE short sentence only.
- Do NOT repeat the original question structure.
- Do NOT say sorry or apologise.
- Keep it conversational.
`

	user := fmt.Sprintf("Original question: %s\nCandidate said: %s", orig, st.LastUser)

	ctx2, cancel := context.WithTimeout(ctx, 12*time.Second)
	defer cancel()

	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.3,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})

	if err != nil || len(resp.Choices) == 0 {
		return "Let me rephrase that: " + orig
	}

	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

func sampleLines(lines []string, n int) string {
	if len(lines) == 0 || n <= 0 {
		return "(none)"
	}
	if len(lines) > n {
		lines = lines[:n]
	}
	for i := range lines {
		lines[i] = "- " + lines[i]
	}
	return strings.Join(lines, "\n")
}

var neutralAcks = []string{
	"Alright.",
	"I understand.",
	"Okay.",
	"Good to know.",
	"Got it.",
	"That makes sense.",
	"Thanks for sharing that.",
}

func pickNeutralAck() string {
	return neutralAcks[rand.Intn(len(neutralAcks))]
}

// Emotion-aware acknowledgement (empathetic when needed, otherwise neutral professional)
func (a *App) generateAcknowledgement(ctx context.Context, st *InterviewState) string {
	ans := strings.TrimSpace(st.LastUser)
	if ans == "" {
		return pickNeutralAck()
	}

	// Emotion → soft empathy
	if isEmotional(ans) {
		return softEmpathyAcks[rand.Intn(len(softEmpathyAcks))]
	}

	// SHORT answers → cheap neutral ack
	if len(ans) < 25 {
		return pickNeutralAck()
	}

	// LLM-based acknowledgement with strict restrictions
	sys := `
You are a recruiter.

Return ONE very short acknowledgement to the candidate's answer.

RULES:
- Max 1 short sentence.
- Do NOT summarize or repeat the candidate’s answer.
- Do NOT mention the specific content or topic of the answer.
- Neutral professional tone.
- No follow-up questions.
- No emojis.
`
	user := fmt.Sprintf("Candidate answer: %s\nLast question asked: %s", ans, st.LastQuestion)

	ctx2, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.2,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return pickNeutralAck()
	}
	ack := strings.TrimSpace(resp.Choices[0].Message.Content)
	if ack == "" {
		return pickNeutralAck()
	}
	return ack
}

func isEmotionalEnding(msg string) bool {
	m := strings.ToLower(msg)
	return strings.Contains(m, "hope") ||
		strings.Contains(m, "nervous") ||
		strings.Contains(m, "worried") ||
		strings.Contains(m, "did i do well") ||
		strings.Contains(m, "not confident") ||
		strings.Contains(m, "anxious")
}

func (a *App) generateClosingMessage(ctx context.Context, st *InterviewState) string {
	if a.LLM == nil {
		return "Thank you for your time today. This concludes the interview and we will follow up with you soon."
	}

	sys := `
You are a professional recruiter. Your job is to close the interview politely.

STRICT RULES:
- 2–3 sentences maximum.
- Professional and warm.
- No repeated phrases (“Noted”, “Understood”, “Thanks for sharing”) unless really natural.
- No filler, no clichés.
- If the interview felt emotional, include a gentle reassurance.
- Do NOT ask any new questions.
- No emojis.
`
	user := fmt.Sprintf(`
Field: %s
Subfield: %s
LastUserMessage: %s
SummaryOfQuestions: %v
`,
		st.Field,
		st.Subfield,
		st.LastUser,
		lastN(st.Asked, 8),
	)

	ctx2, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.3,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return "Thank you for your time. We will stay in touch with the next steps."
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

func (a *App) maybeFollowUp(ctx context.Context, st *InterviewState) string {
	if a.LLM == nil {
		return ""
	}
	if isConfused(st.LastUser) {
		return ""
	}
	if len(st.LastUser) < 40 {
		return ""
	}
	// Very rare follow-ups (5%)
	if rand.Float64() > 0.05 {
		return ""
	}

	sys := `
You are a professional interviewer.

Task:
Write ONE follow-up question that stays on the same topic as the original question.

STRICT RULES:
- Max 9 words.
- One sentence.
- Same topic ONLY.
- No generic 'tell me more'.
- No repeating earlier questions.
- No re-asking anything from the last 3 questions.
- No filler, no fluff.
`
	user := fmt.Sprintf(`
LastQuestion: %s
CandidateAnswer: %s
Previous3: %v
`, st.LastQuestion, st.LastUser, lastN(st.Asked, 3))

	ctx2, cancel := context.WithTimeout(ctx, 12*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.25,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return ""
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

func lastN(arr []string, n int) []string {
	if len(arr) <= n {
		return arr
	}
	return arr[len(arr)-n:]
}

func (a *App) generateCVTailoredQuestion(ctx context.Context, st *InterviewState, qType string) string {
	if a.LLM == nil {
		return ""
	}
	// No CV → skip
	if len(st.Skills) == 0 && len(st.Projects) == 0 && len(st.Experiences) == 0 {
		return ""
	}

	sys := `
You are a senior human interviewer generating a CV-based question.

GOALS:
- Ask ONE creative, fresh, and unique question based on the candidate’s CV.
- Rotate topics: skills → project → experience → challenge → decision-making → teamwork → impact.
- Explore a NEW angle every time.

STRICT RULES:
- Max 2 sentences.
- Must be different in structure and topic from the last 4 questions.
- Do NOT reuse similar verbs, patterns, or templates.
- Do NOT repeat any concept asked earlier.
- Do NOT ask generic "describe/explain/tell me".
- No filler, no acknowledgment.

STYLE:
- Human, conversational, professional.
- Curious but not interrogative.
- Realistic questions a real interviewer would ask.
`

	user := fmt.Sprintf(`
CANDIDATE STRUCTURE:
Field: %s
Subfield: %s
Skills: %v
Projects: %v
Experiences: %v

LAST_QUESTIONS (avoid all topics & patterns):
%v

YOUR TASK:
Generate ONE fresh, novel CV-based question exploring a different aspect of the candidate’s background.
Do not repeat previous verbs, tone, or sentence structure.
Do not ask about the same project twice in a row.
`,
		st.Field, st.Subfield,
		st.Skills,
		st.Projects,
		st.Experiences,
		lastN(st.Asked, 4),
	)

	ctx2, cancel := context.WithTimeout(ctx, 40*time.Second)
	defer cancel()
	resp, err := a.LLM.CreateChatCompletion(ctx2, openai.ChatCompletionRequest{
		Model:       a.ChatModel,
		Temperature: 0.4,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sys},
			{Role: openai.ChatMessageRoleUser, Content: user},
		},
	})
	if err != nil || len(resp.Choices) == 0 {
		return ""
	}
	out := strings.TrimSpace(resp.Choices[0].Message.Content)
	if out == "" {
		return ""
	}
	return out
}

func lastThree(arr []string) []string {
	if len(arr) <= 3 {
		return arr
	}
	return arr[len(arr)-3:]
}

/* -------------------- Evaluation handlers unchanged -------------------- */

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

	// This old endpoint still works; left intact.
	q := "next interview question for " + in.Domain
	qEmb, err := a.embed(r.Context(), []string{q})
	if err != nil || len(qEmb) == 0 {
		http.Error(w, "embedding error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	queryEmb := qEmb[0]

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

	if a.LLM == nil {
		q := "Based on your background, describe how you would approach a key challenge in " + in.Domain + "."
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(nextQResp{Question: q})
		return
	}

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
