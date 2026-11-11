package tensor

import "fmt"

func Fused1614Log2MatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Log2()
	return r, nil
}
func Fused1614Log2MatmulLog2Name() string { return "log2_matmul_log2" }
