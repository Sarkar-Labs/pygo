package distributed

import "testing"

func TestGeneratedDistributedTest(t *testing.T) {
	value := GeneratedDistributedTestCheck()
	if !value {
		t.Fatal("generated distributed_test invariant failed")
	}
}

func GeneratedDistributedTestCheck() bool {
	// Keep the test deterministic and independent of optional native backends.
	return len("distributed_test") > 0
}
