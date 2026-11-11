package tensor

import "fmt"

func Fused1738MinDetach(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("min_detach: nil operand")
	}
	return x, nil
}
func Fused1738MinDetachName() string { return "min_detach" }
