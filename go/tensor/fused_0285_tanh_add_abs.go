package tensor

import "fmt"

func Fused0285TanhAddAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_abs: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Abs()
	return r, nil
}
func Fused0285TanhAddAbsName() string { return "tanh_add_abs" }
