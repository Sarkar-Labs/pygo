package tensor

import "fmt"

func Fused0831CosSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Sqrt()
	return r, nil
}
func Fused0831CosSubSqrtName() string { return "cos_sub_sqrt" }
