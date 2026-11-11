package tensor

import "fmt"

func Fused1029CeilMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Sin()
	return r, nil
}
func Fused1029CeilMatmulSinName() string { return "ceil_matmul_sin" }
