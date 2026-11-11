package tensor

import "fmt"

func Fused0052CeilMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Ceil()
	return r, nil
}
func Fused0052CeilMulName() string { return "ceil_mul" }
