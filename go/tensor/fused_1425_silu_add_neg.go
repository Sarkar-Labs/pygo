package tensor

import "fmt"

func Fused1425SiluAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Neg()
	return r, nil
}
func Fused1425SiluAddNegName() string { return "silu_add_neg" }
