package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Config struct{ data map[string]interface{} }

func New() *Config { return &Config{data: make(map[string]interface{})} }
func FromJSON(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c := New()
	return c, json.NewDecoder(f).Decode(&c.data)
}
func FromEnv(prefix string) *Config {
	c := New()
	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) == 2 && strings.HasPrefix(parts[0], prefix) {
			c.data[strings.ToLower(strings.TrimPrefix(parts[0], prefix+"_"))] = parts[1]
		}
	}
	return c
}
func (c *Config) Get(k string) interface{}    { return c.data[k] }
func (c *Config) Set(k string, v interface{}) { c.data[k] = v }
func (c *Config) GetString(k, def string) string {
	if v, ok := c.data[k].(string); ok {
		return v
	}
	return def
}
func (c *Config) GetInt(k string, def int) int {
	if v, ok := c.data[k].(float64); ok {
		return int(v)
	}
	return def
}
func (c *Config) GetBool(k string, def bool) bool {
	switch v := c.data[k].(type) {
	case bool:
		return v
	case string:
		b, _ := strconv.ParseBool(v)
		return b
	}
	return def
}
func (c *Config) SaveJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(c.data)
}
func (c *Config) String() string {
	b, _ := json.MarshalIndent(c.data, "", "  ")
	return fmt.Sprintf("Config(%s)", string(b))
}
func Merge(base, override *Config) *Config {
	out := New()
	for k, v := range base.data {
		out.data[k] = v
	}
	for k, v := range override.data {
		out.data[k] = v
	}
	return out
}
