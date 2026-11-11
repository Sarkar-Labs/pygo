package tensor

import "fmt"

func Fused1301GeluSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Cos()
	return r, nil
}
func Fused1301GeluSubCosName() string { return "gelu_sub_cos" }
