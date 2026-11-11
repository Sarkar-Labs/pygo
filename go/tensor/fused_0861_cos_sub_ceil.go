package tensor

import "fmt"

func Fused0861CosSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Ceil()
	return r, nil
}
func Fused0861CosSubCeilName() string { return "cos_sub_ceil" }
