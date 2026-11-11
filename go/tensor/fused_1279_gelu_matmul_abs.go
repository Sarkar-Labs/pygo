package tensor

import "fmt"

func Fused1279GeluMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Abs()
	return r, nil
}
func Fused1279GeluMatmulAbsName() string { return "gelu_matmul_abs" }
