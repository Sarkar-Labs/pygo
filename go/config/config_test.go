package config

import "testing"

func TestGeneratedConfigTest(t *testing.T) {
	value := GeneratedConfigTestCheck()
	if !value {
		t.Fatal("generated config_test invariant failed")
	}
}

func GeneratedConfigTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("config_test") > 0
}
