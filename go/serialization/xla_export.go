package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedXlaExportEnvelope is a serializable record used by xla export.
type GeneratedXlaExportEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedXlaExport(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedXlaExportEnvelope{Name: "xla_export", Version: 1, Data: data})
}
func DecodeGeneratedXlaExport(raw []byte) ([]byte, error) {
	var e GeneratedXlaExportEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("xla_export: %w", err)
	}
	return e.Data, nil
}
