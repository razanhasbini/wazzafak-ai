package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

/* ---------------------------------------------------
   DATA STRUCTURES
--------------------------------------------------- */

type QuestionObj struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Difficulty  string `json:"difficulty"`
	Question    string `json:"question"`
	Subcategory string `json:"subcategory"`
}

type FileWrapper struct {
	Major       string        `json:"major"`
	Subcategory string        `json:"subcategory"`
	Questions   []QuestionObj `json:"questions"`
}

/*
QuestionSet = ALL questions grouped by category
main.go expects:

	Technical   []string
	Experience  []string
	Personality []string
	Scenario    []string
	Behavioral  []string
	Common      []string
*/
type QuestionSet struct {
	Technical   []string `json:"technical"`
	Experience  []string `json:"experience"`
	Personality []string `json:"personality"`
	Scenario    []string `json:"scenario"`
	Behavioral  []string `json:"behavioral"`
	Common      []string `json:"common"`
}

type DatasetManager struct {
	Data map[string]QuestionSet
}

/* ---------------------------------------------------
   LOADING ALL JSON DATA
--------------------------------------------------- */

func LoadDatasets() (*DatasetManager, error) {
	root := "./interview_datasets"

	dm := &DatasetManager{
		Data: make(map[string]QuestionSet),
	}

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}
		if !strings.HasSuffix(info.Name(), ".json") {
			return nil
		}

		folder := filepath.Base(filepath.Dir(path))      // e.g. "software"
		file := strings.TrimSuffix(info.Name(), ".json") // e.g. "backend"
		key := folder + "_" + file                       // e.g. software_backend

		raw, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", path, err)
		}

		// Try two formats:
		// FORMAT A (common folder)   → FileWrapper
		// FORMAT B (domain folders)  → []QuestionObj
		var wrap FileWrapper
		var list []QuestionObj

		qs := QuestionSet{}

		// Try A
		if errA := json.Unmarshal(raw, &wrap); errA == nil && wrap.Questions != nil {
			for _, q := range wrap.Questions {
				assignToCategory(&qs, q)
			}
		} else {
			// Try B
			if errB := json.Unmarshal(raw, &list); errB == nil && len(list) > 0 {
				for _, q := range list {
					assignToCategory(&qs, q)
				}
			} else {
				return fmt.Errorf("invalid JSON format in %s", path)
			}
		}

		dm.Data[key] = qs
		return nil
	})

	if err != nil {
		return nil, err
	}

	rand.Seed(time.Now().UnixNano())
	fmt.Printf("📚 Loaded %d dataset categories\n", len(dm.Data))
	return dm, nil
}

/* ---------------------------------------------------
   CATEGORY MAPPING
--------------------------------------------------- */

func assignToCategory(qs *QuestionSet, q QuestionObj) {
	t := strings.ToLower(strings.TrimSpace(q.Type))

	switch t {
	case "technical":
		qs.Technical = append(qs.Technical, q.Question)
	case "experience":
		qs.Experience = append(qs.Experience, q.Question)
	case "personality":
		qs.Personality = append(qs.Personality, q.Question)
	case "scenario":
		qs.Scenario = append(qs.Scenario, q.Question)
	case "behavioral":
		qs.Behavioral = append(qs.Behavioral, q.Question)
	default:
		// anything unknown → Common
		qs.Common = append(qs.Common, q.Question)
	}
}

/* ---------------------------------------------------
   UTIL FUNCTIONS
--------------------------------------------------- */

func pickN(slice []string, n int) []string {
	if len(slice) == 0 {
		return []string{}
	}
	rand.Shuffle(len(slice), func(i, j int) {
		slice[i], slice[j] = slice[j], slice[i]
	})
	if n > len(slice) {
		n = len(slice)
	}
	return slice[:n]
}

/* ---------------------------------------------------
   BUILD INTERVIEW SET
--------------------------------------------------- */

func (dm *DatasetManager) BuildInterview(field string) []string {
	results := []string{}

	// Find dataset key matching field (fuzzy)
	var selected QuestionSet
	for k, v := range dm.Data {
		if strings.Contains(k, strings.ToLower(field)) {
			selected = v
			break
		}
	}

	// Add domain questions
	results = append(results, pickN(selected.Technical, 3)...)
	results = append(results, pickN(selected.Experience, 2)...)
	results = append(results, pickN(selected.Personality, 1)...)
	results = append(results, pickN(selected.Scenario, 1)...)
	results = append(results, pickN(selected.Behavioral, 1)...)

	// Add common/global questions
	commonPers := dm.find("common_personality")
	commonExp := dm.find("common_experience")
	commonSc := dm.find("common_scenario")
	commonBeh := dm.find("common_behavioral")

	results = append(results, pickN(commonPers.Personality, 1)...)
	results = append(results, pickN(commonExp.Experience, 1)...)
	results = append(results, pickN(commonSc.Scenario, 1)...)
	results = append(results, pickN(commonBeh.Behavioral, 1)...)

	return results
}

func (dm *DatasetManager) find(name string) QuestionSet {
	if v, ok := dm.Data[name]; ok {
		return v
	}
	return QuestionSet{}
}

/* ---------------------------------------------------
   SAMPLE QUESTIONS
--------------------------------------------------- */

func (dm *DatasetManager) SampleQuestions(key string, qType string, n int) []string {
	qs, ok := dm.Data[key]
	if !ok {
		return []string{}
	}

	switch strings.ToLower(qType) {
	case "technical":
		return pickN(qs.Technical, n)
	case "experience":
		return pickN(qs.Experience, n)
	case "personality":
		return pickN(qs.Personality, n)
	case "scenario":
		return pickN(qs.Scenario, n)
	case "behavioral":
		return pickN(qs.Behavioral, n)
	case "common":
		return pickN(qs.Common, n)
	default:
		// return full mix
		all := append([]string{}, qs.Technical...)
		all = append(all, qs.Experience...)
		all = append(all, qs.Personality...)
		all = append(all, qs.Scenario...)
		all = append(all, qs.Behavioral...)
		all = append(all, qs.Common...)
		return pickN(all, n)
	}
}
