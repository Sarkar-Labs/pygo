package tensor

import "fmt"

func Fused0851CosSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Cos()
	return r, nil
}
func Fused0851CosSubCosName() string { return "cos_sub_cos" }
