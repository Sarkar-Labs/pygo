package tensor

import "fmt"

func Fused1735MinSqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("min_squeeze: nil operand")
	}
	return x, nil
}
func Fused1735MinSqueezeName() string { return "min_squeeze" }
