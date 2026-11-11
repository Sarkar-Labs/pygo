package tensor

import "fmt"

func Fused1130RoundAddCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_ceil: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Ceil()
	return r, nil
}
func Fused1130RoundAddCeilName() string { return "round_add_ceil" }
