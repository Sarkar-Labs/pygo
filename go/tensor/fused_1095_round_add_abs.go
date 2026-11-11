package tensor

import "fmt"

func Fused1095RoundAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Abs()
	return r, nil
}
func Fused1095RoundAddAbsName() string { return "round_add_abs" }
