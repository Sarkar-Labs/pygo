package tensor

import "fmt"

func Fused1581Log2SubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Ceil()
	return r, nil
}
func Fused1581Log2SubCeilName() string { return "log2_sub_ceil" }
