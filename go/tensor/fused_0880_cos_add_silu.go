package tensor

import "fmt"

func Fused0880CosAddSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_silu: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.SiLU()
	return r, nil
}
func Fused0880CosAddSiluName() string { return "cos_add_silu" }
