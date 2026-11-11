package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedTorchscriptEnvelope is a serializable record used by torchscript.
type GeneratedTorchscriptEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedTorchscript(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedTorchscriptEnvelope{Name: "torchscript", Version: 1, Data: data})
}
func DecodeGeneratedTorchscript(raw []byte) ([]byte, error) {
	var e GeneratedTorchscriptEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("torchscript: %w", err)
	}
	return e.Data, nil
}
