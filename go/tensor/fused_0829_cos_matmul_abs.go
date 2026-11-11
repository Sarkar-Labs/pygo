package tensor

import "fmt"

func Fused0829CosMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Abs()
	return r, nil
}
func Fused0829CosMatmulAbsName() string { return "cos_matmul_abs" }
