package tensor

import "fmt"

func Fused1551Log2SubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Sqrt()
	return r, nil
}
func Fused1551Log2SubSqrtName() string { return "log2_sub_sqrt" }
