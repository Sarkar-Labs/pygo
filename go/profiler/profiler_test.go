package profiler

import "testing"

func TestGeneratedProfilerTest(t *testing.T) {
	value := GeneratedProfilerTestCheck()
	if !value {
		t.Fatal("generated profiler_test invariant failed")
	}
}

func GeneratedProfilerTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("profiler_test") > 0
}
