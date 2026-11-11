package serialization

import (
	"encoding/json"
	"fmt"
)

// GeneratedParquetEnvelope is a serializable record used by parquet.
type GeneratedParquetEnvelope struct {
	Name    string `json:"name"`
	Version int    `json:"version"`
	Data    []byte `json:"data"`
}

func EncodeGeneratedParquet(data []byte) ([]byte, error) {
	return json.Marshal(GeneratedParquetEnvelope{Name: "parquet", Version: 1, Data: data})
}
func DecodeGeneratedParquet(raw []byte) ([]byte, error) {
	var e GeneratedParquetEnvelope
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, fmt.Errorf("parquet: %w", err)
	}
	return e.Data, nil
}
