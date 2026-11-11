package tensor

import "fmt"

func Fused1032CeilMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	r = r.Cos()
	return r, nil
}
func Fused1032CeilMulCosName() string { return "ceil_mul_cos" }
