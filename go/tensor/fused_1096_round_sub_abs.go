package tensor

import "fmt"

func Fused1096RoundSubAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_abs: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Abs()
	return r, nil
}
func Fused1096RoundSubAbsName() string { return "round_sub_abs" }
