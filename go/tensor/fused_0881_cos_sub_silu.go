package tensor

import "fmt"

func Fused0881CosSubSilu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_silu: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.SiLU()
	return r, nil
}
func Fused0881CosSubSiluName() string { return "cos_sub_silu" }
