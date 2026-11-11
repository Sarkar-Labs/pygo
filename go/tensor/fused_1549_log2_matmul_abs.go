package tensor

import "fmt"

func Fused1549Log2MatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Abs()
	return r, nil
}
func Fused1549Log2MatmulAbsName() string { return "log2_matmul_abs" }
