package tensor

import "fmt"

func Fused0711LogSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Log2()
	return r, nil
}
func Fused0711LogSubLog2Name() string { return "log_sub_log2" }
