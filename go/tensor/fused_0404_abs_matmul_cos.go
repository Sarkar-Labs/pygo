package tensor

import "fmt"

func Fused0404AbsMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Cos()
	return r, nil
}
func Fused0404AbsMatmulCosName() string { return "abs_matmul_cos" }
