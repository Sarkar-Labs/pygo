package tensor

import "fmt"

func Fused1280GeluAddSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_sqrt: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Sqrt()
	return r, nil
}
func Fused1280GeluAddSqrtName() string { return "gelu_add_sqrt" }
