package tensor

import "fmt"

func Fused1027CeilMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Sin()
	return r, nil
}
func Fused1027CeilMulSinName() string { return "ceil_mul_sin" }
