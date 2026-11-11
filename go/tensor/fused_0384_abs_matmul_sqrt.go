package tensor

import "fmt"

func Fused0384AbsMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Sqrt()
	return r, nil
}
func Fused0384AbsMatmulSqrtName() string { return "abs_matmul_sqrt" }
