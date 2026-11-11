package tensor

import "fmt"

func Fused0504SqrtMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Ceil()
	return r, nil
}
func Fused0504SqrtMatmulCeilName() string { return "sqrt_matmul_ceil" }
