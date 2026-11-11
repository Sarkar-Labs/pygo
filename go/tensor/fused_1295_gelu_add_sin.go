package tensor

import "fmt"

func Fused1295GeluAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Sin()
	return r, nil
}
func Fused1295GeluAddSinName() string { return "gelu_add_sin" }
