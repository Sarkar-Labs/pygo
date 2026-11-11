package tensor

import "fmt"

func Fused0444AbsMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Log2()
	return r, nil
}
func Fused0444AbsMatmulLog2Name() string { return "abs_matmul_log2" }
