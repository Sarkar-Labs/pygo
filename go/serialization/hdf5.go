package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedHdf5Envelope is a serializable record used by hdf5.
type GeneratedHdf5Envelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedHdf5(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedHdf5Envelope{Name: "hdf5", Version: 1, Data: data})
}
func DecodeGeneratedHdf5(raw []byte) ([]byte, error) {
	var e GeneratedHdf5Envelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("hdf5: %w", err)
	}
	return e.Data, nil
}
