package tensor

import "fmt"

func Fused1131RoundSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Ceil()
	return r, nil
}
func Fused1131RoundSubCeilName() string { return "round_sub_ceil" }
