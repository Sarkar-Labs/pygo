package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedZarrEnvelope is a serializable record used by zarr.
type GeneratedZarrEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedZarr(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedZarrEnvelope{Name: "zarr", Version: 1, Data: data})
}
func DecodeGeneratedZarr(raw []byte) ([]byte, error) {
	var e GeneratedZarrEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("zarr: %w", err)
	}
	return e.Data, nil
}
