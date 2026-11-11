package tensor

import "fmt"

func Fused1009CeilMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Abs()
	return r, nil
}
func Fused1009CeilMatmulAbsName() string { return "ceil_matmul_abs" }
