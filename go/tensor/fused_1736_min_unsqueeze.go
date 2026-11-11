package tensor

import "fmt"

func Fused1736MinUnsqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("min_unsqueeze: nil operand")
	}
	return x, nil
}
func Fused1736MinUnsqueezeName() string { return "min_unsqueeze" }
