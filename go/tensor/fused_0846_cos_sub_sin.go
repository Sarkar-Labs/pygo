package tensor

import "fmt"

func Fused0846CosSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Sin()
	return r, nil
}
func Fused0846CosSubSinName() string { return "cos_sub_sin" }
