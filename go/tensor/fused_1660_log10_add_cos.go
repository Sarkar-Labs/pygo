package tensor

import "fmt"

func Fused1660Log10AddCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_add_cos: nil operand")
	}
	r := x.Add(y)
	r = r.Log10()
	r = r.Cos()
	return r, nil
}
func Fused1660Log10AddCosName() string { return "log10_add_cos" }
