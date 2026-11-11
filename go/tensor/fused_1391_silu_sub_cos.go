package tensor

import "fmt"

func Fused1391SiluSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Cos()
	return r, nil
}
func Fused1391SiluSubCosName() string { return "silu_sub_cos" }
