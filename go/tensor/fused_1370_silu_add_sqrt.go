package tensor

import "fmt"

func Fused1370SiluAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Sqrt()
	return r, nil
}
func Fused1370SiluAddSqrtName() string { return "silu_add_sqrt" }
