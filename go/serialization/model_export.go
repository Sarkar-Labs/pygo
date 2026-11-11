package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedModelExportEnvelope is a serializable record used by model export.
type GeneratedModelExportEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedModelExport(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedModelExportEnvelope{Name: "model_export", Version: 1, Data: data})
}
func DecodeGeneratedModelExport(raw []byte) ([]byte, error) {
	var e GeneratedModelExportEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("model_export: %w", err)
	}
	return e.Data, nil
}
