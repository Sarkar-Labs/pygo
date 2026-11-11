package tensor

import "fmt"

func Fused1390SiluAddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Cos()
	return r, nil
}
func Fused1390SiluAddCosName() string { return "silu_add_cos" }
