package tensor

import "fmt"

func Fused1661Log10SubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Log10()
	r = r.Cos()
	return r, nil
}
func Fused1661Log10SubCosName() string { return "log10_sub_cos" }
