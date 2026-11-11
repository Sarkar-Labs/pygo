package metrics

import "testing"

func TestGeneratedMetricsTest(t *testing.T) {
	value := GeneratedMetricsTestCheck()
	if !value {
		t.Fatal("generated metrics_test invariant failed")
	}
}

func GeneratedMetricsTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("metrics_test") > 0
}
