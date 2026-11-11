package tensor

import "fmt"

func Fused1031CeilSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Ceil()
	r = r.Cos()
	return r, nil
}
func Fused1031CeilSubCosName() string { return "ceil_sub_cos" }
